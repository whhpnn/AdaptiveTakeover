## MPC  Parameter Settings

The MPC controller is designed to track the predicted human intended trajectory. The cost function consists of four terms: the tracking error in the `x` coordinate, the tracking error in the `y` coordinate, the steering input magnitude, and the steering increment.

Specifically:

- `q1`: weight of the tracking error in the `x` coordinate;
- `q2`: weight of the tracking error in the `y` coordinate;
- `r`: weight of the steering input magnitude;
- `g`: weight of the steering increment.

No strict constraints are imposed on the position states. The steering angle is limited to `[-0.5, 0.5]` rad, and the prediction horizon is set to `4`.

The key MPC parameters for different tasks and vehicle speed conditions are listed below.

| Task   | Speed (m/s) | q1 | q2 | r | g |
|--------|------------:|---:|---:|--:|--:|
| Task A | 5           | 6  | 4  | 1 | 0.5 |
| Task A | 8           | 5  | 3  | 2 | 2   |
| Task A | 10          | 8  | 4  | 5 | 3   |
| Task B | 5           | 4  | 4  | 2 | 2   |
| Task B | 8           | 5  | 5  | 3 | 3   |
| Task B | 10          | 4  | 4  | 6 | 6   |
