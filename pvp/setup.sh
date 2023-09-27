tag=$1
url=$2
key=$3
policy_store_dir=$4
name=$5
nmmo_evaluate_root_dir=$6

cd ${nmmo_evaluate_root_dir}
base_dir=$PWD

echo "Cur dir is ${base_dir}"
echo "git clone --no-checkout ${url} ${key}"
git clone --no-checkout ${url} ${key}

echo "cd ${key}"
cd ${key}
echo "Cur dir is ${PWD}"

echo "git checkout ${tag}"
git checkout ${tag}

echo "my-submission"
cd my-submission
echo "Cur dir is ${PWD}"

echo "ls | grep [.]pt | xargs -I file cp file ${policy_store_dir}/${key}.pt"
ls | grep [.]pt | xargs -I file cp file ${policy_store_dir}/${key}.pt

cd ${base_dir}
echo "Cur dir is ${PWD}"
