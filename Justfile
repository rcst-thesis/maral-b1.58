# 80% of system RAM, exec:java runs inside Maven's JVM so heap must be
# set here via MAVEN_OPTS, not inside the pom's <commandlineArgs>.

export MAVEN_OPTS := "-Xmx7372m"

install:
    mvn clean install -Dmaven.test.skip

train:
    mvn clean compile exec:java@train

chart:
    mvn compile exec:java -Dexec.mainClass="com.rcst.TrainingChart" -Dexec.args="checkpoints/training_log.csv checkpoints/training_chart"

# Interactive translation REPL — loads checkpoints/best by default.
# Pass a specific checkpoint as the first argument:

# just infer checkpoints/epoch-028
infer ckpt="checkpoints/best":
    mvn compile exec:java@infer -Dinfer.ckpt={{ ckpt }}

test:
    mvn clean test

tokenize:
    python scripts/train_spm.py

ci: test train
