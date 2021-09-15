import QtQuick 2.10
import QtQuick.Controls 2.3
import QtQuick.Layouts 1.3

Grid {
    width: parent.width
    height: parent.height
    rows: 3
    columns: 2
    rowSpacing: 2
    columnSpacing: 2

    Rectangle {
        width: parent.width / 2 - 1
        height: parent.height / 2 - 1
        color: "red"
    }

    Rectangle {
        width: parent.width / 2 - 1
        height: parent.height / 2 - 1
        color: "#9BCC29"
    }

    Rectangle {
        width: parent.width / 2 - 1
        height: parent.height / 2 - 1
        color: "#9BCC29"
    }

    Rectangle {
        width: parent.width / 2 - 1
        height: parent.height / 2 - 1
        color: "#9BCC29"
    }

}
