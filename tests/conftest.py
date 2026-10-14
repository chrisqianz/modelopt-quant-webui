def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="重写 golden/*.json（只在故意改行为时用，改完看 diff）")
