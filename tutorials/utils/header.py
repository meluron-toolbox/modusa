from IPython.display import HTML, display

def header(
    title, 
    description="", 
    image_path="https://raw.githubusercontent.com/meluron/assets/refs/heads/main/logos/meluron/icon.png",
    link_url="https://github.com/meluron", 
    title_color="#343a40", desc_color="#6c757d", 
    bg_color="#ffffff", border_color="#e9ecef",
    shadow_color="rgba(0,0,0,0.06)", icon_shadow="rgba(231, 76, 60, 0.3)",
    icon_size=50, top_position=7, right_position=15
):
    """
    Create a beautiful title box with clickable icon for Jupyter notebooks.
    
    Parameters:
    -----------
    title : str
        Main title text
    description : str  
        Description text below title
    image_path : str
        Path to the icon image (relative or absolute)
    link_url : str
        URL to link when icon is clicked
    title_color : str, optional
        Color of the title text (default: "#343a40")
    desc_color : str, optional
        Color of the description text (default: "#6c757d")
    bg_color : str, optional
        Background color of the box (default: "#ffffff")
    border_color : str, optional
        Border color (default: "#e9ecef")
    shadow_color : str, optional
        Box shadow color (default: "rgba(0,0,0,0.06)")
    icon_shadow : str, optional
        Icon shadow color (default: "rgba(231, 76, 60, 0.3)")
    icon_size : int, optional
        Size of the icon in pixels (default: 50)
    top_position : int, optional
        Top position of icon in pixels (default: 7)
    right_position : int, optional
        Right position of icon in pixels (default: 15)
    
    Returns:
    --------
    Displays the HTML title box in Jupyter notebook
    """
    
    html_content = f'''
    <div style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 14px 18px;
        margin: 12px 0;
        position: relative;
        box-shadow: 0 2px 10px {shadow_color};
    ">
        <a href="{link_url}" target="_blank" style="
            position: absolute;
            top: 50%;
            right: {right_position}px;
            transform: translateY(-50%);
            text-decoration: none;
            transition: transform 0.2s ease;
        " onmouseover="this.style.transform='translateY(-50%) scale(1.15)'" onmouseout="this.style.transform='translateY(-50%) scale(1)'">
            <img src="{image_path}" alt="Icon" style="
                width: {icon_size}px;
                height: {icon_size}px;
                border-radius: 50%;
                border: 2px solid white;
                box-shadow: 0 2px 6px {icon_shadow};
            ">
        </a>
        <h2 style="margin: 0; color: {title_color}; font-family: 'Segoe UI', sans-serif; font-size: 1.3em;">{title}</h2>
        <p style="margin: 3px 0 0 0; color: {desc_color}; font-size: 0.9em;">{description}</p>
    </div>
    '''
    
    display(HTML(html_content))