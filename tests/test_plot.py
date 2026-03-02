"""Plotting-related unit tests."""
# pylint: disable=protected-access

from control_toolbox import compute_metrics, plot_response


def test_plot_response_basic(simple_step_response):
    """Ensure basic figure structure and title behaviour."""
    fig = plot_response(simple_step_response)
    # one output + one input => two axes
    assert len(fig.get_axes()) == 2
    assert "Closed-loop response" in fig._suptitle.get_text()
    # test custom title
    fig2 = plot_response(simple_step_response, title="my title")
    assert fig2._suptitle.get_text() == "my title"


def test_plot_response_with_metrics(simple_step_response):
    """Plotting with metrics should not change layout and should annotate."""
    # compute simple metrics to trigger annotations
    m = compute_metrics(simple_step_response)
    fig = plot_response(simple_step_response, metrics=m)
    # axes still same number
    assert len(fig.get_axes()) == 2
