# Unreleased

- The units used in River have been corrected to be based on powers of 2 (KiB, MiB). This only changes the display, the behaviour is unchanged.

## tree

- Instead of letting trees grow indefinitely, setting the `max_depth` parameter to `None` will stop the trees from growing when they reach the system recursion limit.
