FROM docker.m.daocloud.io/library/python@sha256:1d52838af602b4b5a831beb13a0e4d073280665ea7be7f69ce2382f29c5a613f
COPY wheels /wheels
COPY scientific_runtime.json /opt/scientific_runtime.json
RUN python -c 'import json,hashlib,pathlib; m=json.load(open("/opt/scientific_runtime.json")); assert all(hashlib.sha256((pathlib.Path("/wheels")/n).read_bytes()).hexdigest()==h for n,h in m["wheels"].items())' \
 && pip install --no-cache-dir --no-index --find-links=/wheels numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 matplotlib==3.7.5 sympy==1.12 python-dateutil==2.9.0.post0 \
 && rm -rf /wheels
ENV OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
WORKDIR /tmp
USER 65534:65534
CMD ["python", "-I", "-u", "-"]
