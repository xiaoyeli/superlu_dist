# Shared machinery of .ci_tests.sh (CPU) and .ci_tests_gpu.sh (GPU).
# This file is sourced, not executed.
#
# A test is one line of the calling script:
#
#     add_test <expect> <check> <np> "<env settings>" <command ...>
#
#   expect  pass    the test must pass
#           xfail   known library defect, tracked: the test is run and a
#                   failure is reported as XFAIL without failing the suite.
#                   If it passes it is reported as XPASS, a reminder to turn
#                   the entry back into "pass".
#   check   how the outcome is read
#           ci      p?test_ci: exit status and its "SLU_CI_RESULT:" line
#           xtrue   an EXAMPLE driver: every "||X - Xtrue|| / ||X||" it prints
#                   must be below $SLU_CI_XTRUE_TOL, and there must be one
#           xtrue_s the same for a single precision driver
#                   ($SLU_CI_XTRUE_TOL_SINGLE)
#           pdtest  TEST/pdtest: its "All tests ... passed the threshold" line
#           berr    a batched EXAMPLE driver: every "Berr = " must be small
#           status  exit status only
#   np      number of MPI processes
#   env     environment of the run, e.g. "GPU3DVERSION=1 SUPERLU_ACC_OFFLOAD=1"
#
#           skip    the test is listed but not run (reported as SKIP); used
#                   for what the machine at hand cannot run
#
# Tests are numbered in the order they are added, and belong to the group
# opened by the last "ci_group <name>" line.
#
# Selection (first argument of the calling script, or $TEST_NUMBER):
#     all | list | N | N-M | <group> | any blank separated mix of these
#
# Environment knobs:
#   SLU_CI_BUILD_DIR   build directory                   (default: ./build)
#   SLU_CI_MPIRUN      launcher up to the process count, e.g. "srun -n"
#                      (default: "mpirun -n", plus --oversubscribe and, for
#                      root, --allow-run-as-root when mpirun is Open MPI)
#   SLU_CI_TIMEOUT     seconds allowed for one test        (default: 300)
#   SLU_CI_XTRUE_TOL   tolerance of the "xtrue"/"berr" checks (default: 1e-6)
#   SLU_CI_XTRUE_TOL_SINGLE  the same in single precision  (default: 1e-2)
#   SLU_CI_VERBOSE     1: print the output of every test, not only of failures

ROOT_DIR="${ROOT_DIR:-$PWD}"
BUILD_DIR="${SLU_CI_BUILD_DIR:-$ROOT_DIR/build}"
DATA_FOLDER="$ROOT_DIR/EXAMPLE"
EXAMPLE_FOLDER="$BUILD_DIR/EXAMPLE"
TEST_FOLDER="$BUILD_DIR/TEST"
# short names for the test tables
D="$DATA_FOLDER"; E="$EXAMPLE_FOLDER"; T="$TEST_FOLDER"

SLU_CI_TIMEOUT="${SLU_CI_TIMEOUT:-300}"
SLU_CI_XTRUE_TOL="${SLU_CI_XTRUE_TOL:-1e-6}"
SLU_CI_XTRUE_TOL_SINGLE="${SLU_CI_XTRUE_TOL_SINGLE:-1e-2}"

if [ -z "${SLU_CI_MPIRUN:-}" ]; then
    SLU_CI_MPIRUN="mpirun"
    # Open MPI: 4.x reports "Open MPI"/"OpenRTE", 5.x may report only its
    # launcher, "prterun (PRRTE)"; ompi_info is installed with all of them.
    if mpirun --version 2>&1 | grep -qiE "open ?mpi|open-rte|openrte|prte|prrte" ||
       command -v ompi_info > /dev/null 2>&1; then
        SLU_CI_MPIRUN="$SLU_CI_MPIRUN --oversubscribe"
        [ "$(id -u)" = "0" ] && SLU_CI_MPIRUN="$SLU_CI_MPIRUN --allow-run-as-root"
    fi
    SLU_CI_MPIRUN="$SLU_CI_MPIRUN -n"
fi

if [ -t 1 ]; then
    C_RED="\033[31;1m"; C_GREEN="\033[32;1m"; C_BLUE="\033[34;1m"; C_OFF="\033[0m"
else
    C_RED=""; C_GREEN=""; C_BLUE=""; C_OFF=""
fi

CI_NTESTS=0
CI_CUR_GROUP=misc
CI_GROUPS=""
declare -a CI_EXPECT CI_CHECK CI_NP CI_ENV CI_CMD CI_GROUP

ci_group() {
    CI_CUR_GROUP="$1"
    CI_GROUPS="$CI_GROUPS $1"
}

add_test() {
    CI_NTESTS=$((CI_NTESTS + 1))
    CI_GROUP[$CI_NTESTS]="$CI_CUR_GROUP"
    CI_EXPECT[$CI_NTESTS]="$1"; CI_CHECK[$CI_NTESTS]="$2"
    CI_NP[$CI_NTESTS]="$3";     CI_ENV[$CI_NTESTS]="$4"
    shift 4
    CI_CMD[$CI_NTESTS]="$*"
}

# Largest value printed after the given text (blanks are not significant);
# empty if the text never appears.
ci_max_after() {  # <file> <text>
    awk -v key="$2" '
        BEGIN { gsub(/[ \t]/, "", key) }
        { s = $0; gsub(/[ \t]/, "", s); p = index(s, key); if (!p) next
          s = substr(s, p + length(key))
          v = (s ~ /^[-+]?[0-9.]/) ? s + 0 : 1e300   # nan, inf: fail
          if (n == 0 || v > m) m = v; n++ }
        END { if (n) printf "%.6e\n", m }' "$1"
}

# Outcome of one run: prints PASS, FAIL or SKIP.
ci_outcome() {  # <check> <exit status> <log>
    local check="$1" status="$2" log="$3" v tol
    case "$check" in
    ci)
        if grep -q "^SLU_CI_RESULT: SKIP" "$log"; then echo SKIP
        elif [ "$status" -eq 0 ] && grep -q "^SLU_CI_RESULT: PASS" "$log"; then echo PASS
        else echo FAIL; fi ;;
    xtrue|xtrue_s|berr)
        tol="$SLU_CI_XTRUE_TOL"
        [ "$check" = xtrue_s ] && tol="$SLU_CI_XTRUE_TOL_SINGLE"
        if [ "$check" = berr ]; then v=$(ci_max_after "$log" "Berr =")
        else v=$(ci_max_after "$log" "||X-Xtrue||/||X|| ="); fi
        if [ "$status" -eq 0 ] && [ -n "$v" ] &&
           awk "BEGIN { exit !($v <= $tol) }"; then echo PASS
        else echo FAIL; fi ;;
    pdtest)
        if [ "$status" -eq 0 ] && grep -q "passed the threshold" "$log"; then echo PASS
        else echo FAIL; fi ;;
    *)
        if [ "$status" -eq 0 ]; then echo PASS; else echo FAIL; fi ;;
    esac
}

CI_NPASS=0; CI_NFAIL=0; CI_NSKIP=0; CI_NXFAIL=0; CI_NXPASS=0
CI_FAILED=""; CI_XPASSED=""

run_test() {  # <number>
    local n="$1" log status outcome label
    local envs="${CI_ENV[$n]}" cmd="${CI_CMD[$n]}" np="${CI_NP[$n]}"
    log=$(mktemp "${TMPDIR:-/tmp}/slu_ci.XXXXXX")

    printf "${C_BLUE}[%3d] %s${C_OFF}\n" "$n" "${envs:+$envs }$SLU_CI_MPIRUN $np ${cmd//$ROOT_DIR\//}"
    if [ "${CI_EXPECT[$n]}" = skip ]; then
        : > "$log"; status=0; outcome=SKIP
    else
        # shellcheck disable=SC2086
        env $envs timeout "$SLU_CI_TIMEOUT" $SLU_CI_MPIRUN "$np" $cmd > "$log" 2>&1
        status=$?
        outcome=$(ci_outcome "${CI_CHECK[$n]}" "$status" "$log")
        [ "$status" -eq 124 ] && echo "      timed out after $SLU_CI_TIMEOUT s" >> "$log"
    fi

    label="$outcome"
    if [ "${CI_EXPECT[$n]}" = xfail ]; then
        [ "$outcome" = FAIL ] && label=XFAIL
        [ "$outcome" = PASS ] && label=XPASS
    fi
    case "$label" in
    PASS)  CI_NPASS=$((CI_NPASS + 1)) ;;
    SKIP)  CI_NSKIP=$((CI_NSKIP + 1)) ;;
    XFAIL) CI_NXFAIL=$((CI_NXFAIL + 1)) ;;
    XPASS) CI_NXPASS=$((CI_NXPASS + 1)); CI_XPASSED="$CI_XPASSED $n" ;;
    FAIL)  CI_NFAIL=$((CI_NFAIL + 1)); CI_FAILED="$CI_FAILED $n" ;;
    esac

    if [ "$label" = FAIL ] || [ "${SLU_CI_VERBOSE:-0}" = 1 ]; then
        sed 's/^/      | /' "$log"
        [ "$label" = FAIL ] && echo "      exit status $status"
    else
        # the lines that carry the verdict
        grep -E "FAILED|cannot run|ERROR|Xtrue\|\| ?/ ?\|\|X|passed the threshold|Berr =" "$log" |
            head -n 12 | sed 's/^ */      | /'
    fi
    if [ "$label" = FAIL ]; then printf "      ${C_RED}%s${C_OFF}\n" "$label"
    else printf "      ${C_GREEN}%s${C_OFF}\n" "$label"; fi
    rm -f "$log"
}

ci_main() {  # <selection>
    local sel="${1:-all}" n tok list=""
    if [ "$sel" = list ]; then
        for n in $(seq 1 $CI_NTESTS); do
            printf "%3d  %-9s %-5s np=%-2s %s%s\n" "$n" "${CI_GROUP[$n]}" "${CI_EXPECT[$n]}" \
                   "${CI_NP[$n]}" "${CI_ENV[$n]:+${CI_ENV[$n]} }" "${CI_CMD[$n]//$ROOT_DIR\//}"
        done
        echo "groups:$CI_GROUPS"
        return 0
    fi

    for tok in $sel; do
        case "$tok" in
        all)
            list="$list $(seq 1 $CI_NTESTS)" ;;
        *[!0-9-]*)   # a group name
            n=""
            for n in $(seq 1 $CI_NTESTS); do
                [ "${CI_GROUP[$n]}" = "$tok" ] && list="$list $n"
            done
            case " $CI_GROUPS " in
            *" $tok "*) ;;
            *) list="$list $tok" ;;   # reported below as unknown
            esac ;;
        *-*) list="$list $(seq "${tok%-*}" "${tok#*-}")" ;;
        *)   list="$list $tok" ;;
        esac
    done

    for n in $list; do
        if ! [ "$n" -ge 1 ] 2>/dev/null || [ "$n" -gt "$CI_NTESTS" ]; then
            printf "${C_RED} ###SLU: Unknown test %s (%d tests; groups:%s)${C_OFF}\n" \
                   "$n" "$CI_NTESTS" "$CI_GROUPS"
            CI_NFAIL=$((CI_NFAIL + 1)); CI_FAILED="$CI_FAILED $n"
            continue
        fi
        run_test "$n"
    done

    echo "------------------------------------------------------------"
    echo "SLU CI summary: $CI_NPASS passed, $CI_NFAIL failed, $CI_NSKIP skipped," \
         "$CI_NXFAIL known failures, $CI_NXPASS unexpected passes"
    [ -n "$CI_FAILED" ] && echo "  failed:$CI_FAILED"
    [ -n "$CI_XPASSED" ] && echo "  unexpected passes (make these 'pass'):$CI_XPASSED"
    [ "$CI_NFAIL" -eq 0 ]
}
