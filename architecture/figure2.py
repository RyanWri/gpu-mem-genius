from diagrams import Diagram, Edge
from diagrams.generic.storage import Storage
from diagrams.aws.ml import Sagemaker
from diagrams.aws.ml import DeepLearningAmis
from diagrams.alibabacloud.security import AntiDdosPro
from diagrams.azure.analytics import AnalysisServices
from diagrams.azure.network import VirtualNetworkClassic

with Diagram(
    "RL Components",
    show=False,
    direction="LR",  # Left to right for a clean row
    graph_attr={"labelloc": "t", "fontsize": "26"},
    filename="figure2",
):
    states = AnalysisServices("env States")
    actions = VirtualNetworkClassic("env Actions")
    buffer = Storage("Replay Buffer")
    agent = Sagemaker("Agent")
    network_params = DeepLearningAmis("Neural Network")
    optimizer = AntiDdosPro("optimizer")

    (
        states
        >> Edge(style="invis")
        >> actions
        >> Edge(style="invis")
        >> buffer
        >> Edge(style="invis")
        >> agent
        >> Edge(style="invis")
        >> network_params
        >> Edge(style="invis")
        >> optimizer
    )
