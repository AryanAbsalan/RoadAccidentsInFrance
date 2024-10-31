from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

# Team member data
team_members = [
    {
        "name": "Aryan Absalan",
        "github": "https://github.com/AryanAbsalan",
        "email": "Aryan.absalan@gmail.com",
    },
    {
        "name": "Jannis Zeelen",
        "github": "https://github.com/JannisZeelen",
        "email": "",
    },
    {
        "name": "Tanja Schroeder",
        "github": "https://github.com/tanjaldir",
        "email": "",
    },
]

# Home page endpoint
@app.get("/", response_class=HTMLResponse)
def homepage() -> HTMLResponse:
    # Constructing the team member list as HTML
    team_member_html = ""
    for member in team_members:
        team_member_html += f"""
        <li>
            <b>{member['name']}</b><br>
            GitHub: <a href="{member['github']}">{member['github']}</a><br>
            Email: {member['email'] if member['email'] else 'Not provided'}
        </li>
        """

    html_content = f"""
    <html>
        <head>
            <title>Welcome to the API</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f4f4f9;
                    color: #333;
                }}
                .container {{
                    text-align: left;
                    background-color: #ffffff;
                    padding: 30px;
                    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                    border-radius: 8px;
                    width: 80%;
                    max-width: 800px;
                }}
                h1 {{
                    color: #2c3e50;
                    font-size: 2em;
                    margin-bottom: 10px;
                }}
                p {{
                    font-size: 1.1em;
                    margin-bottom: 20px;
                }}
                ul {{
                    list-style-type: none;
                    padding: 0;
                }}
                ul li {{
                    background-color: #ecf0f1;
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                }}
                ul li b {{
                    color: #3498db;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Welcome To The Road Accidents In France API</h1>
                <p>This API provides the following endpoints:</p>
                <ul>
                    <li><b>/docs</b> - FastAPI documentation</li>
                    <li><b>/register</b> - Register a new user</li>
                    <li><b>/token</b> - Obtain an authentication token</li>
                    <li><b>/users/me</b> - Retrieve information about the current user</li>
                    <li><b>/predict</b> - Make a prediction</li>
                </ul>
                <h2>Team Members</h2>
                <ul>
                    {team_member_html}
                </ul>
            </div>
        </body>
    </html>
    """

    return HTMLResponse(content=html_content, status_code=200)