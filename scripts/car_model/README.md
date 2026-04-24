# Car Model Scripts

This directory contains scripts whose primary ownership is the standalone `car model` task.

Shared implementation note:

- these scripts may still call Python entrypoints under `ss3dm_prior`
- that reflects shared infrastructure only
- it does not mean the `car model` task belongs to the SS3DM scope

Current key entrypoints:

- `pre_process_car.sh`
- `train_meshfleet_car_v4_trio.sh`
- `eval_meshfleet_car_v4_trio.sh`
