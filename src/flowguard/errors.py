class TransientError(Exception):
    """A failure worth retrying later, such as a timeout, a rate limit or an unusable model answer."""
