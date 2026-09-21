"""
Theme constants and stylesheets for the Control Panel.

Defines colors, fonts, and CSS-like stylesheets for PyQt5 widgets.
"""

# Color palette
COLORS = {
    # Backgrounds
    'background': '#2c2c2c',           # Dark grey panel background
    'background_light': '#3c3c3c',     # Section backgrounds
    'background_darker': '#1e1e1e',    # Input field backgrounds
    'border': '#4a4a4a',               # Border color

    # Text
    'text_primary': '#ffffff',         # White text
    'text_secondary': '#b0b0b0',       # Grey text
    'text_stats': '#3498db',           # Blue for statistics values

    # Action colors
    'accent_positive': '#2ecc71',      # Green (call/check/+EV)
    'accent_negative': '#e74c3c',      # Red (fold/-EV)
    'accent_neutral': '#3498db',       # Blue (raise/bet)
    'accent_warning': '#f39c12',       # Orange/yellow warnings

    # Card colors
    'card_red': '#e74c3c',             # Hearts/Diamonds
    'card_black': '#2c3e50',           # Clubs/Spades

    # Status colors
    'status_active': '#27ae60',        # Active/running
    'status_idle': '#95a5a6',          # Idle/waiting
    'status_error': '#c0392b',         # Error state

    # Button states
    'button_primary': '#3498db',       # Primary button
    'button_primary_hover': '#2980b9', # Primary hover
    'button_success': '#27ae60',       # Success/start button
    'button_success_hover': '#229954', # Success hover
    'button_danger': '#e74c3c',        # Danger/stop button
    'button_danger_hover': '#c0392b',  # Danger hover
}

# Font configuration
FONTS = {
    'family': 'Segoe UI, Arial, sans-serif',
    'family_mono': 'Consolas, Courier New, monospace',
    'size_small': 10,
    'size_normal': 11,
    'size_large': 14,
    'size_xlarge': 18,
    'size_action': 24,
}

# Stylesheet components
def get_main_window_style() -> str:
    """Main window stylesheet."""
    return f"""
        QMainWindow {{
            background-color: {COLORS['background']};
        }}
        QWidget {{
            background-color: {COLORS['background']};
            color: {COLORS['text_primary']};
            font-family: {FONTS['family']};
            font-size: {FONTS['size_normal']}px;
        }}
    """

def get_group_box_style() -> str:
    """GroupBox section style."""
    return f"""
        QGroupBox {{
            background-color: {COLORS['background_light']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            margin-top: 12px;
            padding: 10px;
            font-weight: bold;
            font-size: {FONTS['size_normal']}px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: {COLORS['text_primary']};
        }}
    """

def get_combo_box_style() -> str:
    """ComboBox/dropdown style."""
    return f"""
        QComboBox {{
            background-color: {COLORS['background_darker']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 5px 10px;
            min-width: 50px;
            font-size: {FONTS['size_normal']}px;
        }}
        QComboBox:hover {{
            border-color: {COLORS['accent_neutral']};
        }}
        QComboBox:focus {{
            border-color: {COLORS['accent_neutral']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {COLORS['text_primary']};
            margin-right: 5px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {COLORS['background_darker']};
            color: {COLORS['text_primary']};
            selection-background-color: {COLORS['accent_neutral']};
            selection-color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
        }}
    """

def get_line_edit_style() -> str:
    """Line edit/input field style."""
    return f"""
        QLineEdit {{
            background-color: {COLORS['background_darker']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 5px 8px;
            font-size: {FONTS['size_normal']}px;
        }}
        QLineEdit:hover {{
            border-color: {COLORS['accent_neutral']};
        }}
        QLineEdit:focus {{
            border-color: {COLORS['accent_neutral']};
        }}
        QLineEdit:disabled {{
            background-color: {COLORS['background']};
            color: {COLORS['text_secondary']};
        }}
    """

def get_push_button_style(variant: str = 'primary') -> str:
    """Push button style with variants: primary, success, danger."""
    colors = {
        'primary': (COLORS['button_primary'], COLORS['button_primary_hover']),
        'success': (COLORS['button_success'], COLORS['button_success_hover']),
        'danger': (COLORS['button_danger'], COLORS['button_danger_hover']),
    }
    bg, bg_hover = colors.get(variant, colors['primary'])

    return f"""
        QPushButton {{
            background-color: {bg};
            color: {COLORS['text_primary']};
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-size: {FONTS['size_normal']}px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background-color: {bg_hover};
        }}
        QPushButton:pressed {{
            background-color: {bg_hover};
        }}
        QPushButton:disabled {{
            background-color: {COLORS['text_secondary']};
            color: {COLORS['background']};
        }}
    """

def get_checkbox_style() -> str:
    """Checkbox style."""
    return f"""
        QCheckBox {{
            color: {COLORS['text_primary']};
            font-size: {FONTS['size_normal']}px;
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {COLORS['border']};
            border-radius: 3px;
            background-color: {COLORS['background_darker']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {COLORS['accent_neutral']};
            border-color: {COLORS['accent_neutral']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {COLORS['accent_neutral']};
        }}
    """

def get_table_style() -> str:
    """Table widget style."""
    return f"""
        QTableWidget {{
            background-color: {COLORS['background_darker']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            gridline-color: {COLORS['border']};
            font-size: {FONTS['size_normal']}px;
        }}
        QTableWidget::item {{
            padding: 5px;
        }}
        QTableWidget::item:selected {{
            background-color: {COLORS['accent_neutral']};
        }}
        QHeaderView::section {{
            background-color: {COLORS['background_light']};
            color: {COLORS['text_primary']};
            border: none;
            border-bottom: 1px solid {COLORS['border']};
            padding: 5px;
            font-weight: bold;
        }}
    """

def get_label_style(variant: str = 'normal') -> str:
    """Label style with variants: normal, stats, heading."""
    styles = {
        'normal': f"""
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: {FONTS['size_normal']}px;
            }}
        """,
        'secondary': f"""
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: {FONTS['size_small']}px;
            }}
        """,
        'stats': f"""
            QLabel {{
                color: {COLORS['text_stats']};
                font-size: {FONTS['size_large']}px;
                font-weight: bold;
            }}
        """,
        'heading': f"""
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: {FONTS['size_large']}px;
                font-weight: bold;
            }}
        """,
    }
    return styles.get(variant, styles['normal'])

def get_status_bar_style() -> str:
    """Status bar style."""
    return f"""
        QStatusBar {{
            background-color: {COLORS['background_darker']};
            color: {COLORS['text_secondary']};
            border-top: 1px solid {COLORS['border']};
        }}
    """

def get_action_frame_style(action_type: str = 'neutral') -> str:
    """Action display frame style based on action type."""
    color_map = {
        'fold': COLORS['accent_negative'],
        'call': COLORS['accent_positive'],
        'check': COLORS['accent_positive'],
        'raise': COLORS['accent_neutral'],
        'bet': COLORS['accent_neutral'],
        'neutral': COLORS['text_secondary'],
    }
    accent = color_map.get(action_type.lower(), COLORS['text_secondary'])

    return f"""
        QFrame {{
            background-color: {COLORS['background_light']};
            border: 2px solid {accent};
            border-radius: 8px;
            padding: 10px;
        }}
    """

def get_scrollbar_style() -> str:
    """Scrollbar style."""
    return f"""
        QScrollBar:vertical {{
            background-color: {COLORS['background_darker']};
            width: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background-color: {COLORS['border']};
            border-radius: 6px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {COLORS['text_secondary']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar:horizontal {{
            background-color: {COLORS['background_darker']};
            height: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {COLORS['border']};
            border-radius: 6px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {COLORS['text_secondary']};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
    """

def get_full_stylesheet() -> str:
    """Combined stylesheet for the entire control panel."""
    return "\n".join([
        get_main_window_style(),
        get_group_box_style(),
        get_combo_box_style(),
        get_line_edit_style(),
        get_checkbox_style(),
        get_table_style(),
        get_status_bar_style(),
        get_scrollbar_style(),
    ])
