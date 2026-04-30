# Dataset Layout

The scripts expect a data root passed with `--data-root` or `MSFFAT_DATA_ROOT`.

```text
<DATA_ROOT>/
  df/
    ClosedWorld/
      NoDef/
        X_train_NoDef.pkl
        y_train_NoDef.pkl
        X_valid_NoDef.pkl
        y_valid_NoDef.pkl
        X_test_NoDef.pkl
        y_test_NoDef.pkl
      WTFPAD/
        X_train_WTFPAD.pkl
        y_train_WTFPAD.pkl
        X_valid_WTFPAD.pkl
        y_valid_WTFPAD.pkl
        X_test_WTFPAD.pkl
        y_test_WTFPAD.pkl
      WalkieTalkie/
        X_train_WalkieTalkie.pkl
        y_train_WalkieTalkie.pkl
        X_valid_WalkieTalkie.pkl
        y_valid_WalkieTalkie.pkl
        X_test_WalkieTalkie.pkl
        y_test_WalkieTalkie.pkl
    OpenWorld/
      NoDef/
        X_train_NoDef.pkl
        y_train_NoDef.pkl
        X_valid_NoDef.pkl
        y_valid_NoDef.pkl
        X_test_Mon_NoDef.pkl
        y_test_Mon_NoDef.pkl
        X_test_Unmon_NoDef.pkl
        y_test_Unmon_NoDef.pkl
  dlwf/
    tor_100w_2500tr.npz
    tor_200w_2500tr.npz
    tor_500w_2500tr.npz
    tor_900w_2500tr.npz
    tor_200w_100tr_time_test3d.npz
    tor_200w_100tr_time_test10d.npz
    tor_200w_100tr_time_test2w.npz
    tor_200w_100tr_time_test4w.npz
    tor_200w_100tr_time_test6w.npz
  2tab/
    train.npz
    valid.npz
    test.npz
  3tab/
    train.npz
    valid.npz
    test.npz
  4tab/
    train.npz
    valid.npz
    test.npz
  5tab/
    train.npz
    valid.npz
    test.npz
```

For sequence inputs, values are expected to be direction symbols:

- `+1`: outgoing Tor cell;
- `-1`: incoming Tor cell;
- `0`: representation-level zero-padding after trace termination.

The symbol `0` is not an observed Tor cell and not injected network padding.

