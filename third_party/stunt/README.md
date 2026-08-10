# Thin STUNT adapter / vendoring hook for ssl_tabular_benchmark.
#
# Upstream: https://github.com/jaehyun513/STUNT
# Pin: e860675d0e390dba5f12eb9fd7bdcdd8d379f012
#
# The benchmark implementation lives in src/models/stunt_method.py and ports
# the core self-generated-task + ProtoNet meta-learning logic to the
# project's fixed splits. Optionally clone the full upstream tree here at
# environment build time:
#
#   git clone https://github.com/jaehyun513/STUNT.git third_party/stunt/upstream
#   git -C third_party/stunt/upstream checkout e860675d0e390dba5f12eb9fd7bdcdd8d379f012
#
# Do not claim paper-identical numbers when using smoke settings or sklearn
# k-means fallback instead of FAISS GPU.
