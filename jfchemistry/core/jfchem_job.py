"""Decorator that wraps @job and automatically pulls a field from the parent class."""

from jobflow.core.job import job


def jfchem_job():
    """Decorator that wraps @job and automatically pulls a field from the parent class.

    Args:
        field_name: Name of the class attribute to access
        **extra_job_kwargs: Additional kwargs to pass to @job
    """

    class DeferredJobDecorator:
        def __init__(self, func):
            self.func = func
            self.field_name = "_output_model"

        def __set_name__(self, owner, name):
            # Get the field value from the class
            field_value = getattr(owner, self.field_name)

            # Apply the @job decorator with the kwargs
            decorated_func = job(
                self.func,
                output_schema=field_value,
                files="files",
                properties="properties",
            )

            # Replace this descriptor with the decorated function
            setattr(owner, name, decorated_func)

    return DeferredJobDecorator
