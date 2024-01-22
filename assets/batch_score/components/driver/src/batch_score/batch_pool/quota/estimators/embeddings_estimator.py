from .dv3_estimator import DV3Estimator


class EmbeddingsEstimator(DV3Estimator):
    """
    Estimator for the /embeddings API.
    """
    def _get_prompt(self, request_obj: any) -> "list[str]":
        """
        Embeddings are always estimated like a batch. Even if the prompt is a string, it is put into a batch of one.
        """
        prompt = request_obj.get("input", None)

        if not prompt:
            raise Exception("Unsupported model input payload: cannot determine input for request cost estimation.")

        if isinstance(prompt, str):
            return [prompt]
        else:
            return prompt

    def estimate_request_cost(self, request_obj: any) -> "int | tuple[int]":
        prompt = self._get_prompt(request_obj)
        try:
            return self.calc_tokens_with_tiktoken(prompt)
        except BaseException as e:
            # Default to return 1 if tiktoken fails
            return 1

    def estimate_response_cost(self, request_obj: any, response_obj: any) -> int:
        return response_obj["usage"]["total_tokens"]
