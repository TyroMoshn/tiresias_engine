from __future__ import annotations

from fastapi import APIRouter, Request
from ..schemas.api import FeedbackRequest, FeedbackResponse, SeenBatchRequest, SeenBatchResponse

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])


def get_engine(request: Request):
    return request.app.state.engine


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest, request: Request):
    """
    Submits or undoes user feedback (like, dislike, hide, undo_like, undo_hide, etc.).
    """
    engine = get_engine(request)
    st = req.signal_type.lower().strip()
    if st.startswith("undo_") or st in ("unlike", "unhide", "undislike", "remove"):
        engine.remove_feedback(
            user_id=req.user_id,
            post_id=req.post_id,
            signal_type=st,
        )
    else:
        engine.record_feedback(
            user_id=req.user_id,
            post_id=req.post_id,
            signal_type=req.signal_type,
        )
    return FeedbackResponse(
        success=True,
        user_id=req.user_id,
        post_id=req.post_id,
        signal_type=req.signal_type,
    )


@router.post("/seen", response_model=SeenBatchResponse)
async def submit_seen(req: SeenBatchRequest, request: Request):
    """
    Submits a batch of post IDs that have been displayed/seen by the user.
    Enables soft decay (x0.3) so user sees fresh content without permanent blocking.
    """
    engine = get_engine(request)
    count = engine.db.record_seen_batch(
        user_id=req.user_id,
        post_ids=req.post_ids,
    )
    return SeenBatchResponse(
        success=True,
        recorded_count=count,
    )
