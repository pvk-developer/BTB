from dask.distributed import Client
from dask_kubernetes import KubeCluster, make_pod_spec


def setup_kubernetes(kube_conf=None):
    if isinstance(kube_conf, dict):
        cluster = KubeCluster.from_dict(kube_conf)
    if isinstance(kube_conf, str):
        cluster = KubeCluster(kube_conf)
    else:
        pod = make_pod_spec(
            image='daskdev/dask:latest',
            memory_limit='1G',
            memory_request='1G',
            cpu_limit=1,
            cpu_request=1,
            env={'EXTRA_PIP_PACKAGES': 'baytune'}
        )

        cluster = KubeCluster(pod)

    cluster.adapt(minimum=1, maximum=20)

    return Client(cluster)
