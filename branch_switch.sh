AOGMANEO_LOCATION="../AOgmaNeo"
PYAOGMANEO_LOCATION="$PWD"

echo -n "Enter branch name to switch to and build: "
read branch_name

echo "Switching to branch $branch_name"

cd $AOGMANEO_LOCATION

git fetch origin $branch_name
git checkout $branch_name
git merge origin/$branch_name

cd build
cmake -DBUILD_SHARED_LIBS=On ..
sudo make install

cd $PYAOGMANEO_LOCATION

git fetch origin $branch_name
git checkout $branch_name
git merge origin/$branch_name

export USE_SYSTEM_AOGMANEO=1

python -m pip install .

echo "Done switching!"
