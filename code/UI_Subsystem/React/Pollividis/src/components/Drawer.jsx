import * as React from 'react';
import Box from '@mui/material/Box';
import Drawer from '@mui/material/Drawer';
import Button from '@mui/material/Button';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import MailIcon from '@mui/icons-material/Mail';
import {ExitToApp, Home, Person} from "@material-ui/icons";
import {Link} from "react-router-dom";
import FolderIcon from "@material-ui/icons/Folder";
import MapIcon from '@mui/icons-material/Map';
import GetAppIcon from "@material-ui/icons/GetApp";
import GroupIcon from "@material-ui/icons/Group";
import FeedbackIcon from "@material-ui/icons/Feedback";
import HelpIcon from "@material-ui/icons/Help";
import MenuIcon from '@mui/icons-material/Menu';
import AnalyticsIcon from '@mui/icons-material/Analytics';

export default function RightDrawer() {

    const [state, setState] = React.useState({
        right: false,
    });

    let url = "/";

    const toggleDrawer = (anchor, open) => (event) => {
        if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) {
            return;
        }

        setState({ ...state, [anchor]: open });
    };

    var default_list = ['Academic Login', 'Analyze Sample','Pollen Map', 'Download Dataset','About Us','Send Feedback','How Pollividis Works'];
    var user_list = ['Profile','Previous Analyses', 'Analyze Sample','Pollen Map', 'Download Dataset','About Us','Send Feedback','How Pollividis Works', 'Logout'];
    if (sessionStorage.getItem("academic_id") != null){
        default_list = user_list;
    }
    const list = (anchor) => (
        <Box
            sx={{ width: anchor === 'top' || anchor === 'bottom' ? 'auto' : 250 }}
            role="presentation"
            onClick={toggleDrawer(anchor, false)}
            onKeyDown={toggleDrawer(anchor, false)}
        >
            <List>
                {
                    default_list.map((text) => (
                    <ListItem button key={text}>
                        <ListItemIcon>
                            {(() => {
                                switch (text) {
                                    case "Academic Login":  {url = "/login"} return <div><Home/></div>;
                                    case "Profile": {url = "/profile"} return <Person/>;
                                    case "Previous Analyses": {url = "/previous_analyses"} return <FolderIcon />;
                                    case "Analyze Sample": {url = "/analyze_sample"} return <AnalyticsIcon />;
                                    case "Pollen Map": {url = "/map"} return <MapIcon />;
                                    case "Download Dataset": {url = "/analyze_sample"} return <GetAppIcon/>;
                                    case "About Us": {url = "/about-us"} return <GroupIcon/>;
                                    case "Send Feedback": {url = "/send_feedback"} return  <FeedbackIcon/>;
                                    case "How Pollividis Works": {url = "/how_pollividis_works"} return  <HelpIcon/>;
                                    case "Logout": {url = "/analyze_sample"} return  <ExitToApp/>;
                                    default:      return <MailIcon />;

                                }
                            })()}
                        </ListItemIcon>
                        <Link style={{ textDecoration: 'none'}} to={url}><Button style={{textAlign:"left",color:'black'}}>{text}</Button></Link>
                    </ListItem>
                ))}
            </List>
        </Box>
    );

    return (
        <div>
            {
                <React.Fragment key={'right'}>
                    <Button onClick={toggleDrawer('right', true)}><MenuIcon style={{ color: 'white' }}/></Button>
                    <Drawer
                        anchor={'right'}
                        open={state['right']}
                        onClose={toggleDrawer('right', false)}
                    >
                        {list('right')}
                    </Drawer>
                </React.Fragment>
            }
        </div>
    );
}
