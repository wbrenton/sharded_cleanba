# Makefile from https://github.com/instadeepai/sebulba/blob/main/Makefile

GIT_REPO?=wbrenton/sharded_cleanba
GIT_BRANCH?=main
CHECKOUT_DIR?=sharded_cleanba

ENV_CONFIG=CLOUDSDK_COMPUTE_ZONE=$(ZONE) CLOUDSDK_CORE_PROJECT=$(PROJECT)
BASE_CMD=${ENV_CONFIG} gcloud alpha compute tpus tpu-vm
BASE_CMD_QUEUED=${ENV_CONFIG} gcloud alpha compute tpus queued-resources
BASE_NAME?=sharded_cleanba
LOCAL_DIR_PATH?=~/root/projects/sharded_cleanba
WORKER=all

# Basic TPU configuration.
PROJECT?=your-gcp-project
ZONE?=us-central2-b
ACCELERATOR_TYPE?=v4-8
RUNTIME_VERSION?=tpu-vm-v4-base
NAME?=$(BASE_NAME)-$(ACCELERATOR_TYPE)

info:
	@echo BASE_NAME=$(BASE_NAME)
	@echo LOCAL_DIR_PATH=$(LOCAL_DIR_PATH)
	@echo PROJECT=$(PROJECT)
	@echo ZONE=$(ZONE)
	@echo ACCELERATOR_TYPE=$(ACCELERATOR_TYPE)
	@echo RUNTIME_VERSION=$(RUNTIME_VERSION)

.PHONY: create_vm
create_vm:
	$(BASE_CMD) create $(NAME) \
		--accelerator-type $(ACCELERATOR_TYPE) \
		--version $(RUNTIME_VERSION) \

PHONY: git_clone
git_clone:
	$(BASE_CMD) ssh $(NAME)  \
		--command="GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git clone -b ${GIT_BRANCH} git@github.com:${GIT_REPO}.git ${CHECKOUT_DIR}" \
		-- -A

.PHONY:
git_pull:
	$(BASE_CMD) ssh $(NAME)  \
		--command="cd ${CHECKOUT_DIR} && git pull" \
		-- -A

.PHONY: 
run_setup:
	$(BASE_CMD) ssh $(NAME)  \
		--command="cd ${CHECKOUT_DIR} && bash tpu.sh" \
		-- -A

.PHONY: connect
connect:
	$(BASE_CMD) ssh $(NAME)

.PHONY: list
list:
	$(BASE_CMD) list

VERBS = start describe stop delete

.PHONY: $(VERBS)
$(VERBS):
	$(BASE_CMD) $@ $(NAME)

.PHONY: delete_q
delete_q:
	$(BASE_CMD_QUEUED) delete \
		$(NAME)-q

.PHONY: run
run:
	$(BASE_CMD) ssh $(NAME) --worker=$(WORKER) --command="$(command)"

.PHONY: copy
copy:
	${BASE_CMD} scp $(NAME):~/$(CHECKOUT_DIR) $(LOCAL_DIR_PATH) --recurse