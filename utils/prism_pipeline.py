from dataclasses import dataclass
from enum import IntEnum


class PrismPhase(IntEnum):
    GEOMETRY_ACQUISITION = 0
    STATS_COLLECTION = 1
    DEAD_PRUNE_ROUND = 2
    CANDIDATE_PRUNE_ROUND = 3
    RECOVERY_FINE_TUNE = 4
    FINAL_FINE_TUNE = 5


@dataclass
class PrismPipelineConfig:
    enabled: bool = False
    geometry_acq_until_iter: int = -1
    stats_collection_iters: int = 500
    dead_rounds: int = 1
    candidate_rounds: int = 3
    recovery_iters: int = 400
    final_finetune_iters: int = 500
    topology_freeze_during_stats: bool = True
    round_checkpoint: bool = True
    post_commit_recollect_iters: int = 0
    force_recompute_scores_after_recollect: bool = True


class PrismRoundController:
    """
    Multi-round PRISM prune-refine state machine.

    This controller is training-loop agnostic:
    - it emits per-iteration scheduling decisions
    - caller reports prune results via report_prune_result()
    """

    def __init__(self, cfg: PrismPipelineConfig, first_iter: int, total_iters: int):
        self.cfg = cfg
        self.first_iter = int(first_iter)
        self.total_iters = int(total_iters)

        self.geometry_acq_until_iter = int(cfg.geometry_acq_until_iter)
        self.stats_start_iter = self.geometry_acq_until_iter + 1
        self.stats_end_iter = self.stats_start_iter + int(max(0, cfg.stats_collection_iters)) - 1

        self.dead_round_done = 0
        self.candidate_round_done = 0

        self.recovery_remaining = 0
        self.post_commit_recollect_remaining = 0
        self._pending_recollect_after_recovery = False
        self._force_recompute_after_recollect = False
        self.phase = PrismPhase.GEOMETRY_ACQUISITION if bool(cfg.enabled) else PrismPhase.FINAL_FINE_TUNE
        self._phase_initialized = False
        self._prune_pending = False
        self._last_phase = self.phase
        self.pruned_this_round = 0
        self.last_counterfactual_accept = 0
        self.last_rollback = 0
        self.last_recollect_iters_used = 0

    def _transition_if_needed(self, iteration: int):
        if not bool(self.cfg.enabled):
            self.phase = PrismPhase.FINAL_FINE_TUNE
            return

        it = int(iteration)
        if it <= self.geometry_acq_until_iter:
            self.phase = PrismPhase.GEOMETRY_ACQUISITION
            return

        if self.recovery_remaining > 0:
            self.phase = PrismPhase.RECOVERY_FINE_TUNE
            return

        if self.post_commit_recollect_remaining > 0:
            self.phase = PrismPhase.STATS_COLLECTION
            return

        if it <= self.stats_end_iter:
            self.phase = PrismPhase.STATS_COLLECTION
            return

        if self.dead_round_done < int(self.cfg.dead_rounds):
            self.phase = PrismPhase.DEAD_PRUNE_ROUND
            return

        if self.candidate_round_done < int(self.cfg.candidate_rounds):
            self.phase = PrismPhase.CANDIDATE_PRUNE_ROUND
            return

        self.phase = PrismPhase.FINAL_FINE_TUNE

    def step(self, iteration: int):
        prev_phase = self.phase
        self._transition_if_needed(iteration)
        phase_changed = self.phase != prev_phase
        if phase_changed:
            self._last_phase = prev_phase
            # Entering a prune phase => allow exactly one prune attempt for that round.
            if self.phase in (PrismPhase.DEAD_PRUNE_ROUND, PrismPhase.CANDIDATE_PRUNE_ROUND):
                self._prune_pending = True
            else:
                self._prune_pending = False

        collect_stats = self.phase in (
            PrismPhase.STATS_COLLECTION,
            PrismPhase.DEAD_PRUNE_ROUND,
            PrismPhase.CANDIDATE_PRUNE_ROUND,
            PrismPhase.RECOVERY_FINE_TUNE,
            PrismPhase.FINAL_FINE_TUNE,
        )
        allow_topology_mutation = True
        if self.phase == PrismPhase.STATS_COLLECTION and bool(self.cfg.topology_freeze_during_stats):
            allow_topology_mutation = False
        if self.phase == PrismPhase.RECOVERY_FINE_TUNE:
            allow_topology_mutation = False

        prune_mode = None
        should_attempt_prune = False
        if self.phase == PrismPhase.DEAD_PRUNE_ROUND:
            prune_mode = "dead"
            should_attempt_prune = bool(self._prune_pending)
        elif self.phase == PrismPhase.CANDIDATE_PRUNE_ROUND:
            prune_mode = "candidate"
            should_attempt_prune = bool(self._prune_pending)

        return {
            "phase": self.phase,
            "collect_stats": bool(collect_stats),
            "allow_topology_mutation": bool(allow_topology_mutation),
            "should_attempt_prune": bool(should_attempt_prune),
            "prune_mode": prune_mode,
            "post_commit_recollect_remaining": int(self.post_commit_recollect_remaining),
        }

    def report_prune_result(self, prune_mode: str, committed: bool, pruned_count: int, counterfactual_accept: int, rollback: int):
        self.pruned_this_round = int(max(0, pruned_count))
        self.last_counterfactual_accept = int(counterfactual_accept)
        self.last_rollback = int(rollback)
        # Consume this round's single prune attempt.
        self._prune_pending = False
        if prune_mode == "dead":
            self.dead_round_done += 1
        elif prune_mode == "candidate":
            self.candidate_round_done += 1

        self.last_recollect_iters_used = 0
        if committed and prune_mode == "candidate":
            rec = int(max(0, getattr(self.cfg, "post_commit_recollect_iters", 0)))
            self.last_recollect_iters_used = rec
            if rec > 0:
                if int(self.cfg.recovery_iters) > 0:
                    self._pending_recollect_after_recovery = True
                else:
                    self.post_commit_recollect_remaining = rec

        if committed and int(self.cfg.recovery_iters) > 0:
            self.recovery_remaining = int(self.cfg.recovery_iters)
            return

        # If no recovery is inserted, schedule the next round attempt (one-shot)
        # on the next iteration until configured round counts are reached.
        if prune_mode == "dead":
            if self.dead_round_done < int(self.cfg.dead_rounds):
                self._prune_pending = True
        elif prune_mode == "candidate":
            if self.candidate_round_done < int(self.cfg.candidate_rounds):
                self._prune_pending = True

    def consume_recovery_step(self):
        if self.recovery_remaining > 0:
            self.recovery_remaining -= 1
            if self.recovery_remaining == 0 and bool(self._pending_recollect_after_recovery):
                rec = int(max(0, getattr(self.cfg, "post_commit_recollect_iters", 0)))
                if rec > 0:
                    self.post_commit_recollect_remaining = rec
                self._pending_recollect_after_recovery = False
            return
        if self.post_commit_recollect_remaining > 0:
            self.post_commit_recollect_remaining -= 1
            if self.post_commit_recollect_remaining == 0 and bool(getattr(self.cfg, "force_recompute_scores_after_recollect", True)):
                self._force_recompute_after_recollect = True

    def consume_force_recompute_flag(self) -> bool:
        flag = bool(self._force_recompute_after_recollect)
        self._force_recompute_after_recollect = False
        return flag
