"""Static project dashboard: every stage publishes panels (PANELS.md); `alexandria dashboard build|serve`."""
from .site import build, make_server, serve, serve_forever_in_thread

__all__ = ["build", "make_server", "serve", "serve_forever_in_thread"]
