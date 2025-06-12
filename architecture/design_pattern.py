from diagrams import Diagram, Cluster, Edge
from diagrams.generic.compute import Rack
from diagrams.programming.flowchart import InputOutput, Action
from diagrams.aws.ml import MachineLearning
from diagrams.onprem.client import User
from diagrams.aws.management import OpsworksMonitoring


def simple_memory_forecast_pattern(
    filename="simple_memory_forecast_pattern", advanced=False
):
    with Diagram(
        "Simple Memory Forecast Design Pattern",
        filename=filename,
        show=False,
        direction="TB",
        graph_attr={"labelloc": "t", "fontsize": "24"},
    ):
        user = User("Engineer")
        # Inputs
        agent_params = MachineLearning("Agent Parameters")
        states = Action("State Representation")
        actions = Action("Action Representation")
        # Forecast Model
        forecast_model = Rack("Forecast Model")
        forecast_table = InputOutput("CPU/GPU Forecast Table")

        # Data flow
        if not advanced:
            (
                user
                >> Edge(color="black", style="dotted")
                >> [agent_params, states, actions]
                >> Edge(color="blue", style="dotted")
                >> forecast_model
                >> forecast_table
            )
            return 1

        with Cluster("Sampling Features (from Training)"):
            gpu_metrics = OpsworksMonitoring("gpu_metrics")
            cpu_metrics = OpsworksMonitoring("cpu_metrics")
            buffer = OpsworksMonitoring("replay_buffer_metrics")

        (
            user
            >> Edge(color="black", style="dotted")
            >> [agent_params, states, actions]
            >> Edge(color="blue", style="dotted")
            >> forecast_model
        )
        (
            user
            >> Edge(color="black", style="dotted")
            >> [gpu_metrics, cpu_metrics, buffer]
            >> Edge(color="brown", style="dotted")
            >> forecast_model
        )
        forecast_model >> Edge(color="green", style="bold") >> forecast_table


if __name__ == "__main__":
    simple_memory_forecast_pattern(
        filename="architecture/images/simple_memory_forecast_pattern", advanced=False
    )
    simple_memory_forecast_pattern(
        filename="architecture/images/advanced_memory_forecast_pattern", advanced=True
    )
