from fastapi import Depends

from app.security import require_api_key


ProtectedRoute = Depends(require_api_key)
