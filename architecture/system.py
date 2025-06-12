from diagrams import Diagram, Cluster, Edge
from diagrams.generic.compute import Rack
from diagrams.programming.flowchart import Database, InputOutput, Action
from diagrams.aws.ml import SagemakerModel, MachineLearning
from diagrams.onprem.client import User

def rl_system_architecture(filename="rl_system_architecture"):
    with Diagram(
        "RL System Architecture",
        filename=filename,
        show=False,
        direction="TB",
        graph_attr={"labelloc": "t", "fontsize": "30"}
    ):
        user = User("Engineer")

        # RL Environment cluster
        with Cluster("RL Environment"):
            states = Action("States")
            actions = Action("Actions")

        # Agent cluster
        with Cluster("Agent"):
            model = SagemakerModel("Model")
            parameters = MachineLearning("Parameters")

        # Replay Buffer cluster
        with Cluster("Replay Buffer"):
            experience = Database("Experience\n(state, action, reward, next_state, done)")

        # Forecast Engine cluster
        with Cluster("Forecast Engine"):
            forecast_model = SagemakerModel("Forecast Model")
            forecast_table = InputOutput("CPU/GPU Forecast Table")

        # Main RL flow
        states >> Edge(label="reward + new_state", color="blue") >> model
        model >> Edge(label="action", color="blue") >> actions
        
        # Experience is saved after environment step
        actions >> Edge(label="save experience", color="darkgreen") >> experience

        # agent parameters
        model >> Edge(color="black", style="bold") >> parameters

        # Metrics to forecast model (dotted brown)
        states >> Edge(style="dotted", color="brown", label="metrics") >> forecast_model
        actions >> Edge(style="dotted", color="brown", label="metrics") >> forecast_model
        parameters >> Edge(style="dotted", color="brown", label="metrics") >> forecast_model
        experience >> Edge(style="dotted", color="brown", label="metrics") >> forecast_model

        # Forecast model output
        forecast_model >> Edge(color="purple") >> forecast_table
        forecast_table >> Edge(color="orange") >> user

if __name__ == "__main__":
    rl_system_architecture(filename="architecture/images/rl_system_architecture")
