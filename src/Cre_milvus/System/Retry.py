from tenacity import retry, stop_after_attempt, wait_fixed


def retry_function(func, max_attempts=3, wait_seconds=2):
    @retry(stop=stop_after_attempt(max_attempts), wait=wait_fixed(wait_seconds))
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # 只有返回None或False才重试
        if result is None or result is False:
            raise RuntimeError("Function failed, triggering retry")
        return result
    return wrapper