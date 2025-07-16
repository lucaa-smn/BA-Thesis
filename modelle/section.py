from abc import ABC, abstractmethod
import dash


class Section(ABC):

    @abstractmethod
    def get_html(self) -> dash.html.Div:
        pass

    @abstractmethod
    def register_callbacks(self) -> None:
        pass
