from selfedit.tasks import load_tasks, run_hidden, run_visible


def main():
    """Check that every reference solution passes its own visible and hidden tests."""
    tasks = load_tasks()
    bad_v, bad_h = [], []
    for t in tasks:
        if not run_visible(t, t.reference).passed:
            bad_v.append(t.task_id)
        if not run_hidden(t, t.reference).passed:
            bad_h.append(t.task_id)
    wrong = "def placeholder(*a, **k):\n    return None\n"
    print("tasks", len(tasks), "visible-fail", bad_v, "hidden-fail", bad_h)
    print("wrong code rejected:", not run_hidden(tasks[0], wrong).passed)


if __name__ == "__main__":
    main()
