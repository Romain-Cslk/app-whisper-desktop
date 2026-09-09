"""Qt palette and restrained styling for both supported themes."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(dark: bool) -> None:
    app = QApplication.instance()
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: '#151923' if dark else '#f3f5f9',
        QPalette.ColorRole.WindowText: '#edf0f7' if dark else '#1f2937',
        QPalette.ColorRole.Base: '#1d2330' if dark else '#ffffff',
        QPalette.ColorRole.AlternateBase: '#252d3d' if dark else '#eef1f7',
        QPalette.ColorRole.Text: '#edf0f7' if dark else '#1f2937',
        QPalette.ColorRole.Button: '#2b3446' if dark else '#e7ebf3',
        QPalette.ColorRole.ButtonText: '#edf0f7' if dark else '#1f2937',
        QPalette.ColorRole.Highlight: '#5369d6',
        QPalette.ColorRole.HighlightedText: '#ffffff',
        QPalette.ColorRole.ToolTipBase: '#252d3d' if dark else '#ffffff',
        QPalette.ColorRole.ToolTipText: '#edf0f7' if dark else '#1f2937',
        QPalette.ColorRole.PlaceholderText: '#a6aec0' if dark else '#667085',
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor('#808897'))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor('#808897'))
    app.setPalette(palette)
    app.setStyleSheet('''
        QWidget { font-size: 10pt; }
        QPushButton { padding: 7px 12px; border-radius: 5px; }
        QPushButton#startButton { background: #5369d6; color: white; font-weight: bold; }
        QPushButton#startButton:disabled { background: #73798b; color: #dddddd; }
        QLineEdit, QComboBox { padding: 5px; min-height: 21px; }
        QGroupBox { font-weight: bold; border: 1px solid rgba(122,130,148,60); border-radius: 7px; margin-top: 10px; padding-top: 12px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QProgressBar { border: 1px solid rgba(122,130,148,80); border-radius: 4px; text-align: center; min-height: 17px; }
        QProgressBar::chunk { background: #5369d6; border-radius: 3px; }
        QTabWidget::pane { border: 0; }
        QTabBar::tab { padding: 10px 16px; }
    ''')
