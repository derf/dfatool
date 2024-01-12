# Assessing Model Quality

## Tabular Comparison

`--show-model-error` outputs a table that shows LUT error, model error, and
static error. In general, the model is suitable if its prediction error is
close to LUT error and far away from static error.

The error metric can be selected via `--error-metric`.

It is generally a good idea to combine `--show-model-error` with
`--cross-validate=kfold:10` and, depending on application,
`--parameter-aware-cross-validation`. Note that LUT error serves as baseline
("lowest achievable prediction error" / underlying measurement uncertainty)
and is always reported without cross-validation.

## LaTeX dataref export

`--export-dref=filename.tex` exports model statistics and all available error
metrics to a [dataref](https://ctan.org/pkg/dataref) file. Again, it may be a
good idea to also specify `--cross-validate=kfold:10` and possibly
`--parameter-aware-cross-validation`.

`--dref-precision` can be used to limit the number of decimals in the exported
values.
