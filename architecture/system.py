from diagrams import Diagram, Cluster, Edge
from diagrams.programming.flowchart import Database, InputOutput, Action
from diagrams.aws.ml import SagemakerModel, MachineLearning
from diagrams.onprem.client import User
from diagrams.onprem.certificates import CertManager


def rl_system_architecture(filename="rl_system_architecture"):
    with Diagram(
        "RL System Architecture",
        filename=filename,
        show=False,
        direction="TB",
        graph_attr={"labelloc": "t", "fontsize": "30"},
    ):
        user = User("Engineer")

        # RL Environment cluster
        with Cluster("RL Environment"):
            states = Action("States")
            actions = Action("Actions")
            env_manager = CertManager("Env Manager")

        # Agent cluster
        with Cluster("Agent"):
            model = SagemakerModel("Model")
            parameters = MachineLearning("Parameters")

        # Replay Buffer cluster
        with Cluster("Replay Buffer"):
            experience = Database(
                "Experience\n(state, action, reward, next_state, done)"
            )

        # Forecast Engine cluster
        with Cluster("Forecast Engine"):
            forecast_model = SagemakerModel("Forecast Model")
            forecast_table = InputOutput("CPU/GPU Forecast Table")

        # Main RL system flow (all dotted for harmony)
        (
            [states, actions]
            >> Edge(label="observation", style="dotted", color="#4682B4")
            >> env_manager
        )
        (
            env_manager
            >> Edge(label="request action", style="dotted", color="#8FBC8F")
            >> model
        )
        model >> Edge(label="action", style="dotted", color="#8FBC8F") >> env_manager

        # Env Manager writes to buffer
        (
            env_manager
            >> Edge(label="write experience", style="dotted", color="#D2691E")
            >> experience
        )
        # Agent reads from buffer
        (
            experience
            << Edge(label="sample batch", style="dotted", color="#B8860B")
            << model
        )

        # Agent and Model
        model >> Edge(label="parameters", style="dotted", color="#708090") >> parameters

        # All components push metrics to forecast model (array-driven)
        components = [env_manager, parameters, experience]
        for c in components:
            c >> Edge(style="dotted", color="brown", label="metrics") >> forecast_model

        # Forecast Engine output
        (
            forecast_model
            >> Edge(style="dotted", color="#9932CC", label="forecast")
            >> forecast_table
        )
        forecast_table >> Edge(style="dotted", color="#FF8C00", label="result") >> user


if __name__ == "__main__":
    rl_system_architecture(filename="architecture/images/rl_system_architecture")
