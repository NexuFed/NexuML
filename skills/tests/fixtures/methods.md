# Synthetic reference protocol: RMS normalization

This is an executable methods fixture, not a publication or paper-parity claim.

For each four-element input vector x, compute

`normalized[i] = x[i] / sqrt(sum(x[j] ** 2 for j in range(4)) / 4 + epsilon)`

Use epsilon = 0.000001 inside the square root. There is no centering, learned
scale, fitted statistic, stochastic operation or training. Preserve the original
features under their input key and emit normalized values under a distinct key.

Reference inputs are [0, 0, 0, 0], [1, 2, 3, 4] and [-1, 1, -1, 1]. Compare
each output against the equation with absolute and relative tolerances of 1e-6.
Verify finite gradients on a differentiable nonzero input. Use the supplied
64-vector dataset only to check routing/loader identity; keep its train/validation
membership fixed and its final-test loader empty. Report synthetic numerical
parity, not reproduction of an empirical publication result.
