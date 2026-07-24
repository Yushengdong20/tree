"""grasp_object 抓取服务 HTTP 客户端。"""

from .grasp_request_errors import NoGraspObjectError


class GraspRequestClient:
    """封装抓取服务请求和基础响应校验。"""

    def _request_grasp_payload(self):
        import requests

        # 关键步骤：新抓取服务通过 query 参数选择类别和 nearest/multi 模式。
        response = requests.get(
            self.grasp_url,
            params={
                "target_class_id": self.target_class_id,
                "mode": self.grasp_mode,
            },
            timeout=self.http_timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not payload.get("success", False):
            error = (
                payload.get("error", "服务返回 success=false")
                if isinstance(payload, dict)
                else payload
            )
            error_text = str(error)
            lowered_error = error_text.lower()
            if (
                "no object" in lowered_error
                or "empty" in lowered_error
                or "没有" in error_text
                or "无目标" in error_text
                or "无抓取" in error_text
                or "无物体" in error_text
            ):
                raise NoGraspObjectError(error_text)
            raise RuntimeError(error_text)
        return payload

