from tagc.web import Server, load_state

if __name__ == "__main__":
    state = load_state(r"data/unstate")
    server = Server(state)
    server.plot()
    server.app.run_server(debug=True)
