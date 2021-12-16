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

export default function TemporaryDrawer() {

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

    const list = (anchor) => (
        <Box
            sx={{ width: anchor === 'top' || anchor === 'bottom' ? 'auto' : 250 }}
            role="presentation"
            onClick={toggleDrawer(anchor, false)}
            onKeyDown={toggleDrawer(anchor, false)}
        >
            <List>
                {['Academic Login', 'Profile', 'Previous Analysis', 'Analyze Sample','Pollen Map', 'Download Dataset','About Us','Send Feedback','How PolliVidis Works','Logout'].map((text, index) => (
                    <ListItem button key={text}>
                        <ListItemIcon>
                            {(() => {
                                switch (index) {
                                    case 0:  {url = "/login"} return <div><Home/></div>;
                                    case 1: {url = "/profile"} return <Person/>;
                                    case 2: {url = "/"} return <FolderIcon />;
                                    case 3: {url = "/"} return <AnalyticsIcon />;
                                    case 4: {url = "/map"} return <MapIcon />;
                                    case 5: {url = "/"} return <GetAppIcon/>;
                                    case 6: {url = "/about-us"} return <GroupIcon/>;
                                    case 7: {url = "/"} return  <FeedbackIcon/>;
                                    case 8: {url = "/"} return  <HelpIcon/>;
                                    case 9: {url = "/"} return  <ExitToApp/>;
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
