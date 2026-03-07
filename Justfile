install:
    mvn clean install -Dmaven.test.skip

dev:
    mvn clean compile exec:java

test:
    mvn clean test

tokenize:
    python scripts/train_spm.py
