"""Utility functions for manipulating nodes."""

from aiida import orm


def get_last_calcjob(workchain: orm.WorkChainNode) -> orm.CalcJobNode:
    """Return the last CalcJob of a WorkChain."""
    calcs = []
    for called_descendant in workchain.called_descendants:
        if not isinstance(called_descendant, orm.CalcJobNode):
            continue
        calcs.append(called_descendant)

    if len(calcs) == 0:
        return None

    # Sort by schedular_job_id to get latest calcjob
    calcs.sort(key=lambda _: int(_.get_job_id()))
    last_calcjob = calcs[-1]

    return last_calcjob
