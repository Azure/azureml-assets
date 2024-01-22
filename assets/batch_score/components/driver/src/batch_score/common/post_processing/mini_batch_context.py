class MiniBatchContext(object):
    def __init__(self, raw_mini_batch_context, target_result_len) -> None:
        self.__target_result_len = target_result_len
        self.raw_mini_batch_context = raw_mini_batch_context
        self.exception = None

    @property
    def mini_batch_id(self):
        return self.raw_mini_batch_context.minibatch_index

    @property
    def target_result_len(self):
        return self.__target_result_len