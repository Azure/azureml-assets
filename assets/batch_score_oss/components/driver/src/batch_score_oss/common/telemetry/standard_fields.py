# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Defines Part B of the logging schema, optional keys that have a common meaning across telemetry data.

The schema is defined in: https://msdata.visualstudio.com/Vienna/_wiki/wikis/Vienna.wiki/4672/Common-Schema.
"""
from enum import IntEnum


class AzureMLTelemetryTaskResult(IntEnum):
    """The task result types of AzureMl telemetry."""

    Success = 1
    Failure = 2
    Cancelled = 3


class AzureMLTelemetryFailureReason(IntEnum):
    """The failure reason types of AzureMl telemetry."""

    UserError = 1
    SystemError = 2


class AzureMLTelemetryComputeType(IntEnum):
    """
    The compute types of AzureMl telemetry.

    # noqa: E501
    Reference:
    https://msdata.visualstudio.com/Vienna/_git/vienna?path=%2Fsrc%2Fazureml-infra%2Fsrc%2FCommon%2FMachineLearning.Common.TelemetryEvents.Abstractions%2FAzureMLTelemetryEventStandardFields.cs&version=GBmaster&line=93&lineEnd=93&lineStartColumn=5&lineEndColumn=51&lineStyle=plain&_a=contents
    """

    # cSpell: disable

    # *** Azure Machine Learning Compute
    AmlcTrain = 1  # Azure machine learning compute Training cluster, aka. BatchAI, Machine learning compute
    AmlcInference = 2  # Azure machine learning compute Inference cluster
    AmlcDsi = 3  # Azure machine learning compute Data science instance

    # *** non AzureML compute, aka attached compute
    # VirtualMachine
    Remote = 4  # use this if you are not sure the type of your attached VM
    AzureDatabricks = 5

    # HDInsight
    HdiCluster = 6
    AKS = 7  # Azure Kubernetes Service
    ADLA = 8  # Azure Data Lake Analytics, attached compute

    # ContainerInstance
    ACI = 9  # Azure Container Instance

    # *** Azure Synapse Sparkpool
    # Microsoft.ProjectArcadia
    Arcadia = 10

    # Microsoft.SparkOnCosmos
    SparkOnCosmos = 11

    # Microsoft.AzureNotebookVM
    AzureNotebookVM = 12

    # *** DSVM rp
    DSVM = 20  # Data Science VM

    # *** BatchAI rp
    BatchAI = 30

    # *** ITP
    ITP = 40  # This is deprecated and replaced by ITPCompute

    # *** ITPCompute
    ITPCompute = 41

    # *** Cmk8s
    Cmk8s = 45

    # *** MIR
    MIR = 48
    MIR_v2 = 49  # used in MIR vnext in BillingUsage

    # *** Local
    Local = 50

    # *** AMLK8sARC
    AMLK8sARC = 90

    # *** AMLK8sAKS
    AMLK8sAKS = 91

    Others = 100
    # cSpell: enable

    @classmethod
    def current(cls) -> int:
        """Return the current compute type."""
        # if IS_CMAKS:
        #     return cls.AMLK8sAKS

        return cls.AmlcTrain


class AzureMLTelemetryOS(IntEnum):
    """The OS types of AzureMl telemetry."""

    Windows = 1
    Linux = 2
    MacOS = 3
    Android = 4
    iOS = 5
    Others = 100


class AzureMLTelemetryDatasetType(IntEnum):
    """The Dataset types of AzureML telemetry."""

    File = 1
    Tabular = 2
    TimeSeries = 3
    Others = 100


class StandardFields:
    """Defines Part B of the logging schema, optional keys that have a common meaning across telemetry data."""

    def __init__(
        self,
        Duration=None,
        TaskResult: AzureMLTelemetryTaskResult = None,
        FailureReason: AzureMLTelemetryFailureReason = None,
        WorkspaceRegion=None,
        ComputeType: AzureMLTelemetryComputeType = None,
        Attribution=None,
        ClientOS: AzureMLTelemetryOS = None,
        ClientOSVersion=None,
        ClientVersion=None,
        RunId=None,
        ParentRunId=None,
        ExperimentId=None,
        NumberOfNodes=None,
        NumberOfCores=None,
        DatasetType: AzureMLTelemetryDatasetType = None,
        CoreSeconds=None,
    ):
        """Initialize a new instance of the StandardFields."""
        # pylint: disable=invalid-name
        self.Duration = Duration
        self.TaskResult = TaskResult
        self.FailureReason = FailureReason
        self.WorkspaceRegion = WorkspaceRegion
        self.ComputeType = ComputeType
        self.Attribution = Attribution
        self.ClientOS = ClientOS
        self.ClientOSVersion = ClientOSVersion
        self.ClientVersion = ClientVersion
        self.RunId = RunId
        self.ParentRunId = ParentRunId
        self.ExperimentId = ExperimentId
        self.NumberOfNodes = NumberOfNodes
        self.NumberOfCores = NumberOfCores
        self.DatasetType = DatasetType
        self.CoreSeconds = CoreSeconds
