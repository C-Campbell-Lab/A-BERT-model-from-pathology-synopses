from tagc.io_utils import load_state
from tagc.web import Server

if __name__ == "__main__":
    state = load_state("data/unstate.pkl")
    server = Server(state)
    server.plot()
    server.app.run_server(debug=True)
