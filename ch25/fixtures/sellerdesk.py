"""Second setting (Chapter 25): a seller-support desk's action dispatcher.

Handlers register themselves by name and are called through a dict lookup or
getattr(), the way plugin systems, web routers and task queues usually work.
"""
HANDLERS = {}


def register(action):
    def wrap(fn):
        HANDLERS[action] = fn
        return fn
    return wrap


@register("refund")
def handle_refund(ticket):
    return f"refund issued for order {ticket['order']}"


@register("reship")
def handle_reship(ticket):
    return f"replacement shipped for order {ticket['order']}"


class Escalations:
    def escalate_fraud(self, ticket):
        return f"fraud team paged for seller {ticket['seller']}"


def dispatch(ticket):
    if ticket["action"] in HANDLERS:
        return HANDLERS[ticket["action"]](ticket)
    return getattr(Escalations(), "escalate_" + ticket["action"])(ticket)
