from .dv3_estimator import DV3Estimator


# Estimator for the /chat/completions API
class ChatCompletionEstimator(DV3Estimator):
    def _get_prompt(self, request_obj: any) -> str:
        messages = request_obj.get("messages", None)
        prompt: str = None

        if messages:
            for message in messages:
                content: str = message.get("content", None)
                if content:
                    prompt = (prompt if prompt else "") + content
        if not prompt:
            raise Exception("Unsupported model input payload: cannot determine prompt for request cost estimation")

        return prompt