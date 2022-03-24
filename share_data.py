class ShareData:
    def __init__(self, _manager):
        self.l_predict_list = _manager.list()
        self.a_predict_list = _manager.list()
        self.b_predict_list = _manager.list()
        self.l_actual_list = _manager.list()
        self.a_actual_list = _manager.list()
        self.b_actual_list = _manager.list()
        self.x_list = _manager.list()
