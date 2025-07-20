from diagrams import Diagram, Cluster, Edge
from diagrams.elastic.agent import Integrations
from diagrams.onprem.monitoring import Prometheus
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.aws.ml import Sagemaker
from diagrams.programming.flowchart import InputOutput
from diagrams.digitalocean.storage import Space

with Diagram(
    "System Architecture: RL Memory Estimation",
    show=False,
    direction="LR",
    graph_attr={"labelloc": "t", "fontsize": "28"},
    filename="figure1"
):

    # Computation Layer (three components)
    with Cluster("Computation Layer", graph_attr={"bgcolor": "#e3f2fd", "color": "#1976d2"}):
        env = Rack("Environment")
        agent = Integrations("Agent")
        buffer = Storage("Replay Buffer")
        computation = [env, agent, buffer]

    # Feature Extraction Layer (single analytics/monitoring node)
    with Cluster("Feature Extraction Layer", graph_attr={"bgcolor": "#f9fbe7", "color": "#689f38"}):
        extractor = Prometheus("Feature Extractor")

    # Forecast Layer (ML model with two output nodes)
    with Cluster("Forecast Layer", graph_attr={"bgcolor": "#e3f2fd", "color": "#1976d2"}):
        forecast = Sagemaker("Forecast Model")
        cpu = Space("CPU")
        gpu = Space("GPU")

    # Data flow: All Computation components → Feature Extractor
    for comp in computation:
        comp >> Edge(label="metrics", color="#4682B4") >> extractor

    # Feature Extractor → Forecast Model
    extractor >> Edge(label="features", color="#8FBC8F") >> forecast

    # Forecast Model → Outputs
    forecast >> Edge(label="predicts", color="#D2691E") >> cpu
    forecast >> Edge(label="predicts", color="#D2691E") >> gpu
