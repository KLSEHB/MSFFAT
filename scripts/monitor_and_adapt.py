"""Alias of the canonical stateful monitor; see --help for the v2 interface.

The old one-interval interface evaluated probes on the entire interval. Both
monitor commands now use exactly the same disjoint splits and decisions.
"""

from simulate_awf_time_deployment import main


if __name__ == "__main__":
    main()
