# Data Analysis and Performance Model Generation from Log Files

Here, dfatool works with lines of the form "`[::]` *Key* | *parameters* | *NFP values*", where *parameters* is a space-separated series of *param=value* entries (i.e., benchmark configuration) and *NFP values* is a space-separate series of *NFP=value* entries (i.e., benchmark output).
All measurements of a given *Key* combination must use the same set of NFP names.
Parameter names may be different -- parameters that are present in other lines of the same *Key* will be treated as undefined in those lines where they are missing.

Use `bin/analyze-log.py file1.txt file2.txt ...` for analysis.
