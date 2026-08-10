# Lock files are generated AFTER the first successful bootstrap on the cluster.
# Place pip freeze / conda env export outputs here:
#   environment/core/lock/conda-lock.yml  (or environment-explicit.yml)
#   environment/core/lock/pip-freeze.txt
#   environment/tfm/lock/conda-lock.yml
#   environment/tfm/lock/pip-freeze.txt
#   environment/tfm/lock/torch-build.txt   # records cuXXX wheel + torch.__version__
#
# Until those exist, treat versions as unpinned and do not claim reproducibility.
placeholder: true
generated: false
note: "Populate via bootstrap_* scripts on first successful run."
