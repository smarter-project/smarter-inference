class AdmissionControllerException(Exception):
    pass


class NoPerfDataFoundException(AdmissionControllerException):
    pass


class DeadlineMissException(AdmissionControllerException):
    pass


class UnsatisfiableRequestException(AdmissionControllerException):
    pass
