# Unreleased

## stream

- stream.iter_arff and stream.iter_libsvm now close files they opened when iteration is closed or parsing fails. Streams passed in by the caller remain open.
