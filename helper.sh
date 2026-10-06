#!/usr/bin/env bash

# ===> HEADER SECTION START  ===>

# http://bash.cumulonim.biz/NullGlob.html
shopt -s nullglob
# -------------------------------
this_folder="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -z "$this_folder" ]; then
  this_folder=$(dirname $(readlink -f $0))
fi
parent_folder=$(dirname "$this_folder")

# -------------------------------
# --- required functions
debug(){
    local __msg="$1"
    echo " [DEBUG] `date` ... $__msg "
}

info(){
    local __msg="$1"
    echo " [INFO]  `date` ->>> $__msg "
}

warn(){
    local __msg="$1"
    echo " [WARN]  `date` *** $__msg "
}

err(){
    local __msg="$1"
    echo " [ERR]   `date` !!! $__msg "
}


source_if_exists() {
  local file="$1"
  if [ ! -f "$file" ]; then
    warn "we DON'T have a $(basename "$file") file - creating it"
    touch "$file"
    chmod 600 "$file"
  else
    . "$file"
  fi
}

# ---------- CONSTANTS ----------
export FILE_VARIABLES=${FILE_VARIABLES:-".variables"}
export FILE_LOCAL_VARIABLES=${FILE_LOCAL_VARIABLES:-".local_variables"}
export FILE_SECRETS=${FILE_SECRETS:-".secrets"}
export INCLUDE_FILE=${INCLUDE_FILE:-"bashutils"}
export BASHUTILS_REPO="tgedr/bashutils"

find_local_release(){
  local local_file="$this_folder/$INCLUDE_FILE"
  local local_version=0
  if [ -f "$local_file" ]; then
    IFS= read -r version_line < "$local_file"
    local_version=${version_line##*: }
    # info "[find_local_release] local version: $local_version"
    if [[ ! $local_version =~ ^[0-9]+$ ]]; then
      err "[find_local_release] version is not a valid integer: $local_version"
      return 1
    fi
  fi
  echo "$local_version"
}

find_latest_release(){
  [ -z $1 ] && err "[find_latest_release] missing argument REPO" && return 1
  local REPO="$1"
  local latest_release
  latest_release=$(gh release view --repo "$REPO" --json tagName --jq .tagName) || return 1
  echo "$latest_release"
}

define_release_to_update(){
  local local_release
  local latest_release
  local_release=$(find_local_release)
  [ "$?" -ne "0" ] && err "[update_bashutils] failed to find local release" && return 1
  latest_release=$(find_latest_release "$BASHUTILS_REPO")
  [ "$?" -ne "0" ] && err "[update_bashutils] failed to find latest release" && return 1
  local result
  if (( local_release < latest_release )); then
    read -r -p "proceed to update? [y/N] " answer
    case "$answer" in
      [Yy])
        result="$latest_release"
        ;;
      *)
        result=-1 # cancelled
        ;;
    esac
  else
    result=-2 # local release is up-to-date
  fi
  echo "$result"
}

update_bashutils(){
  info "[update_bashutils|in] ($1)"

  [ -z "$BASHUTILS_AUTO_UPDATE" ] || [ "$BASHUTILS_AUTO_UPDATE" -ne "1" ] && warn "[update_bashutils] auto update is disabled" && return 1

  local _pwd=$(pwd)
  local release
  local result

  release=$(define_release_to_update)
  [ "$?" -ne "0" ] && err "[update_bashutils] failed to define the release to update" && return 1
  [ "$release" -eq "-1" ] && info "[update_bashutils] no update performed, update cancelled" && return 0
  [ "$release" -eq "-2" ] && info "[update_bashutils] no update performed, local release is up-to-date" && return 0

  # Use assets API with Accept: application/octet-stream to avoid redirect to
  # objects.githubusercontent.com (which may be blocked by proxies like Zscaler)
  local release_json
  release_json=$(eval curl -fsSL "\"https://api.github.com/repos/$BASHUTILS_REPO/releases/tags/$release\"")
  result="$?"
  if [ "$result" -ne "0" ]; then
    err "[update_bashutils] failed to fetch release metadata"
    cd "$_pwd"
    return 1
  fi

  cd "$this_folder" || exit 1

  echo "$release_json" | python3 -c "
import sys, json
assets = json.load(sys.stdin).get('assets', [])
for a in assets:
    print(a['id'], a['name'])
  " | while read -r asset_id asset_name; do
    info "[get_updated_release] downloading asset: $asset_name (id: $asset_id)"
    eval curl -fsSL -H "\"Accept: application/octet-stream\"" \
      -o "\"$asset_name\"" \
      "\"https://api.github.com/repos/$BASHUTILS_REPO/releases/assets/$asset_id\""
    if [ "$?" -ne "0" ]; then
      err "[get_updated_release] failed to download asset: $asset_name"
      cd "$_pwd"
      return 1
    fi
  done
  result="$?"
  cd "$_pwd"

  [ "$result" -ne "0" ] && err "[get_updated_release|out] => ${result}" && return 1
  info "[get_updated_release|out] => ${result}"
}

# -------------------------------
# --- source variables files
source_if_exists "$this_folder/$FILE_VARIABLES"
source_if_exists "$this_folder/$FILE_LOCAL_VARIABLES"
source_if_exists "$this_folder/$FILE_SECRETS"

# ---------- include bashutils ----------
BASHUTILS_AUTO_UPDATE="${BASHUTILS_AUTO_UPDATE:-0}"
update_bashutils
. "$this_folder/$INCLUDE_FILE"

# <=== HEADER SECTION END  <===


# ===> MAIN SECTION    ===>
# ---------- CONSTANTS ----------
export SRC_DIR=${SRC_DIR:-"${this_folder}/src"}
export TEST_DIR=${TEST_DIR:-"${this_folder}/test"}
# -------------------------------
# --- main functions

reqs(){
  info "[reqs|in]"

  which uv
  if [ ! "$?" -eq "0" ]; then 
    info "[reqs] uv is not installed. Installing now...";
    curl -LsSf https://astral.sh/uv/install.sh | sh
    [ ! "$?" -eq "0" ] && err "[reqs] failed to install uv" && exit 1
  fi

  _pwd=$(pwd)
  cd "$this_folder"

  uv sync --group dev
  local result="$?"
  if [ ! "$result" -eq "0" ] ; then err "[reqs] could not install dependencies"; fi

  cd "$_pwd"

  local msg="[reqs|out] => ${result}"
  [[ ! "$result" -eq "0" ]] && info "$msg" && exit 1
  info "$msg"
}


# <=== MAIN SECTION END  <===


# ===> FOOTER SECTION START  ===>

usage() {
  cat <<EOM
  usage:
  $(basename $0) { option }
    options:
      - update_bashutils [VERSION]            updates bashutils to the latest or a specific GitHub release
      - reqs                                  installs development requirements
      - linter_check                          runs code lint and format check
      - sast_check                            runs static application security tests (SAST) check
      - sca_check                             runs software component analysis (SCA) check
      - test [<test_folder>]                  runs unit tests
      - test_coverage                         prints test coverage report
      - test_coverage_check <threshold>       checks coverage against a threshold
      - build                                 builds the package
      - publish                               publishes the package
      - tag                                   creates a git tag with the version and pushes it
EOM
  exit 1
}


case "$1" in
  update_bashutils)
    BASHUTILS_AUTO_UPDATE=1 update_bashutils "$2"
    ;;
  reqs)
    reqs
    ;;
  linter_check)
    lint_check_ruff_uv
    ;;
  sast_check)
    sast_check_bandit_uv "$SRC_DIR"
    ;;
  sca_check)
    sca_check_safety_uv "$SAFETY_KEY"
    ;;
  test)
    pytest_uv "$TEST_DIR" && uv run genbadge coverage -i coverage.xml -o coverage.svg
    ;;
  test_coverage)
    test_print_coverage_uv
    ;;
  test_coverage_check)
    test_coverage_check_uv "$2"
    ;;
  build)
    build_uv
    ;;
  publish)
    publish_pypi_uv "$PYPI_TOKEN"
    ;;
  tag)
    git_tag_and_push_auto_uv
    ;;
  *)
    usage
    ;;
esac

# <=== FOOTER SECTION END  <===
