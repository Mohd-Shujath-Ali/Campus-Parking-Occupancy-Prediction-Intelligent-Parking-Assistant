"""
Rate limiting configuration for LPU Campus Parking Intelligence API.
Protects resource-intensive CV inference and live snapshot endpoints from denial-of-service.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
